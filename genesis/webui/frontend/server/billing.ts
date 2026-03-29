import { Router, raw } from 'express';
import Stripe from 'stripe';
import { requireAuth, getAuth } from '@clerk/express';
import {
  ensureUser,
  getSubscription,
  upsertSubscription,
  getSubscriptionByStripeCustomer,
} from './db.js';

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY!, {
  apiVersion: '2026-03-25.dahlia',
});

const router = Router();

// GET /api/billing/status -- returns current subscription status for the authenticated user
router.get('/status', requireAuth(), async (req, res) => {
  try {
    const { userId } = getAuth(req);
    if (!userId) {
      return res.status(401).json({ error: 'Unauthorized' });
    }

    const user = await ensureUser(userId);
    const subscription = await getSubscription(user.id);

    res.json({
      subscribed: subscription?.status === 'active',
      status: subscription?.status ?? 'none',
      plan: subscription?.plan ?? 'free',
      currentPeriodEnd: subscription?.current_period_end ?? null,
    });
  } catch (error) {
    console.error('[BILLING] Error fetching status:', error);
    res.status(500).json({ error: 'Failed to fetch billing status' });
  }
});

// POST /api/billing/checkout -- creates a Stripe Checkout session
router.post('/checkout', requireAuth(), async (req, res) => {
  try {
    const { userId } = getAuth(req);
    if (!userId) {
      return res.status(401).json({ error: 'Unauthorized' });
    }

    const user = await ensureUser(userId);
    const subscription = await getSubscription(user.id);

    const customerData: Stripe.Checkout.SessionCreateParams.CustomerCreation = 'always';
    const sessionParams: Stripe.Checkout.SessionCreateParams = {
      mode: 'subscription',
      line_items: [
        {
          price: process.env.STRIPE_PRICE_ID!,
          quantity: 1,
        },
      ],
      success_url: `${process.env.APP_URL || 'http://localhost:5173'}/?checkout=success`,
      cancel_url: `${process.env.APP_URL || 'http://localhost:5173'}/?checkout=canceled`,
      metadata: {
        clerk_user_id: userId,
        internal_user_id: String(user.id),
      },
    };

    if (subscription?.stripe_customer_id) {
      sessionParams.customer = subscription.stripe_customer_id;
    } else {
      sessionParams.customer_creation = customerData;
      sessionParams.customer_email = user.email ?? undefined;
    }

    const session = await stripe.checkout.sessions.create(sessionParams);
    res.json({ url: session.url });
  } catch (error) {
    console.error('[BILLING] Error creating checkout session:', error);
    res.status(500).json({ error: 'Failed to create checkout session' });
  }
});

// POST /api/billing/portal -- creates a Stripe Billing Portal session
router.post('/portal', requireAuth(), async (req, res) => {
  try {
    const { userId } = getAuth(req);
    if (!userId) {
      return res.status(401).json({ error: 'Unauthorized' });
    }

    const user = await ensureUser(userId);
    const subscription = await getSubscription(user.id);

    if (!subscription?.stripe_customer_id) {
      return res.status(400).json({ error: 'No billing account found' });
    }

    const session = await stripe.billingPortal.sessions.create({
      customer: subscription.stripe_customer_id,
      return_url: process.env.APP_URL || 'http://localhost:5173',
    });

    res.json({ url: session.url });
  } catch (error) {
    console.error('[BILLING] Error creating portal session:', error);
    res.status(500).json({ error: 'Failed to create portal session' });
  }
});

// POST /api/webhooks/stripe -- Stripe webhook handler (no auth -- verified by signature)
const webhookRouter = Router();

webhookRouter.post(
  '/stripe',
  raw({ type: 'application/json' }),
  async (req, res) => {
    const sig = req.headers['stripe-signature'] as string;

    let event: Stripe.Event;
    try {
      event = stripe.webhooks.constructEvent(
        req.body,
        sig,
        process.env.STRIPE_WEBHOOK_SECRET!,
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      console.error('[WEBHOOK] Signature verification failed:', message);
      return res.status(400).json({ error: `Webhook Error: ${message}` });
    }

    try {
      switch (event.type) {
        case 'checkout.session.completed': {
          const session = event.data.object as Stripe.Checkout.Session;
          const clerkUserId = session.metadata?.clerk_user_id;
          if (!clerkUserId) break;

          const user = await ensureUser(clerkUserId);
          const customerId =
            typeof session.customer === 'string'
              ? session.customer
              : session.customer?.id;

          if (customerId && session.subscription) {
            const subscriptionId =
              typeof session.subscription === 'string'
                ? session.subscription
                : session.subscription.id;

            await upsertSubscription(user.id, {
              stripe_customer_id: customerId,
              stripe_subscription_id: subscriptionId,
              status: 'active',
              plan: 'pro',
            });
          }
          break;
        }

        case 'customer.subscription.updated': {
          const subscription = event.data.object as Stripe.Subscription;
          const customerId =
            typeof subscription.customer === 'string'
              ? subscription.customer
              : subscription.customer.id;

          const existing = await getSubscriptionByStripeCustomer(customerId);
          if (existing) {
            await upsertSubscription(existing.user_id, {
              stripe_customer_id: customerId,
              stripe_subscription_id: subscription.id,
              status: subscription.status,
            });
          }
          break;
        }

        case 'customer.subscription.deleted': {
          const subscription = event.data.object as Stripe.Subscription;
          const customerId =
            typeof subscription.customer === 'string'
              ? subscription.customer
              : subscription.customer.id;

          const existing = await getSubscriptionByStripeCustomer(customerId);
          if (existing) {
            await upsertSubscription(existing.user_id, {
              stripe_customer_id: customerId,
              stripe_subscription_id: subscription.id,
              status: 'canceled',
            });
          }
          break;
        }
      }
    } catch (error) {
      console.error('[WEBHOOK] Error processing event:', error);
      return res.status(500).json({ error: 'Webhook processing failed' });
    }

    res.json({ received: true });
  },
);

export { router as billingRouter, webhookRouter };
